class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.scale = (feature_dim // num_heads) ** -0.5

        Multi-head Query, Key, and Value layers with DWConv for Queries
        self.query_layers = nn.ModuleList(
            [nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim) for _ in range(3)]
        ) 
        self.key_layers = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.value_layers = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])

        # Output projection
        self.out_projection = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, f1, f2, f3):
        # Assume f1, f2, f3 are [batch_size, feature_dim]
        features = [f1, f2, f3]
        attended_features = []
        prev_output = None  # To cascade outputs across heads

        for i in range(3):
            # Prepare queries, keys, and values
            q = self.query_layers[i](features[i].unsqueeze(-1)).squeeze(-1)  
            k = torch.stack([self.key_layers[j](features[j]) for j in range(3)], dim=1)  # Shape: [batch_size, 3, feature_dim]
            v = torch.stack([self.value_layers[j](features[j]) for j in range(3)], dim=1)  # Shape: [batch_size, 3, feature_dim]

            # Add previous head's output (if available) to the query
            if prev_output is not None:
                features[i] = features[i] + prev_output  # Modify input for the next head

            # Reshape for multi-head attention
            q = q.view(features[i].size(0), self.num_heads, -1)
            k = k.view(features[i].size(0), self.num_heads, 3, -1)  # 3 sources
            v = v.view(features[i].size(0), self.num_heads, 3, -1)

            # Compute attention scores
            attention_scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) * self.scale
            attention_probs = F.softmax(attention_scores, dim=-1)

            # Apply dropout to the attention probabilities
            attention_probs = self.dropout(attention_probs)

            # Apply attention to values
            attended = torch.matmul(attention_probs, v).squeeze(2).view(features[i].size(0), -1)

            # Apply output projection
            attended = self.out_projection[i](attended)

            # Store attended features and update the cascading output
            attended_features.append(attended)
            prev_output = attended  # Update for cascading

        return attended_features
