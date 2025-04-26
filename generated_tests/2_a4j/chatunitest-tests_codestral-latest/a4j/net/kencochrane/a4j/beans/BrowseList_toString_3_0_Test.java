package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseList_toString_3_0_Test {

    private BrowseList browseList;

    private ArrayList<BrowseNode> nodes;

    @BeforeEach
    void setUp() {
        browseList = new BrowseList();
        nodes = new ArrayList<>();
    }

    @Test
    void testToStringWithNodes() {
        BrowseNode node1 = mock(BrowseNode.class);
        when(node1.getBrowseName()).thenReturn("Node1");
        when(node1.getBrowseId()).thenReturn("ID1");
        BrowseNode node2 = mock(BrowseNode.class);
        when(node2.getBrowseName()).thenReturn("Node2");
        when(node2.getBrowseId()).thenReturn("ID2");
        nodes.add(node1);
        nodes.add(node2);
        browseList.setBrowseNode(nodes.toArray(new BrowseNode[0]));
        String expected = "# of nodes = 2\n" + "Name: Node1\n" + "ID: ID1\n" + "Name: Node2\n" + "ID: ID2\n";
        assertEquals(expected, browseList.toString());
    }

    @Test
    void testToStringWithNoNodes() {
        String expected = "No nodes\n";
        assertEquals(expected, browseList.toString());
    }
}
