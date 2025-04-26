package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_addSubNode_5_0_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @Mock
    private BrowseNode subNode;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testAddSubNode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodesField.set(browseNode, subNodes);
        // Act
        browseNode.addSubNode(subNode);
        // Assert
        assertEquals(1, subNodes.size());
        assertTrue(subNodes.contains(subNode));
    }
}
