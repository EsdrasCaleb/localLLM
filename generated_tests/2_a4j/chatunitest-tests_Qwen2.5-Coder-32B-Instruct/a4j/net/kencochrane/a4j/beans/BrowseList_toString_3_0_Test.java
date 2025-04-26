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

class BrowseList_toString_3_0_Test {

    @Mock
    private BrowseNode mockBrowseNode1;

    @Mock
    private BrowseNode mockBrowseNode2;

    private BrowseList browseList;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        browseList = new BrowseList();
        when(mockBrowseNode1.getBrowseName()).thenReturn("Node1");
        when(mockBrowseNode1.getBrowseId()).thenReturn("123");
        when(mockBrowseNode2.getBrowseName()).thenReturn("Node2");
        when(mockBrowseNode2.getBrowseId()).thenReturn("456");
    }

    @Test
    void testToString_NoNodes() throws Exception {
        // Arrange
        Field nodesField = BrowseList.class.getDeclaredField("nodes");
        nodesField.setAccessible(true);
        nodesField.set(browseList, new ArrayList<BrowseNode>());
        // Act
        String result = browseList.toString();
        // Assert
        assertEquals("No nodes\n", result);
    }

    @Test
    void testToString_OneNode() throws Exception {
        // Arrange
        Field nodesField = BrowseList.class.getDeclaredField("nodes");
        nodesField.setAccessible(true);
        nodesField.set(browseList, new ArrayList<BrowseNode>() {

            {
                add(mockBrowseNode1);
            }
        });
        // Act
        String result = browseList.toString();
        // Assert
        assertEquals("# of nodes = 1\nName: Node1\nID: 123\n", result);
    }

    @Test
    void testToString_MultipleNodes() throws Exception {
        // Arrange
        Field nodesField = BrowseList.class.getDeclaredField("nodes");
        nodesField.setAccessible(true);
        nodesField.set(browseList, new ArrayList<BrowseNode>() {

            {
                add(mockBrowseNode1);
                add(mockBrowseNode2);
            }
        });
        // Act
        String result = browseList.toString();
        // Assert
        assertEquals("# of nodes = 2\nName: Node1\nID: 123\nName: Node2\nID: 456\n", result);
    }

    @Test
    void testToString_NodeIsNull() throws Exception {
        // Arrange
        Field nodesField = BrowseList.class.getDeclaredField("nodes");
        nodesField.setAccessible(true);
        nodesField.set(browseList, new ArrayList<BrowseNode>() {

            {
                add(null);
            }
        });
        // Act
        String result = browseList.toString();
        // Assert
        assertEquals("# of nodes = 1\n", result);
    }
}
