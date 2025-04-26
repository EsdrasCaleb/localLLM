package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_addSubNode_5_0_Test {

    @Test
    public void testAddSubNode() {
        // Arrange
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseId("Example");
        browseNode.setBrowseName("Example Browse");
        browseNode.setMode("Browse");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        // Act
        browseNode.addSubNode(browseNode);
        // Assert
        Assertions.assertEquals(subNodes.size(), 1, "Subnode should be added");
    }
}
