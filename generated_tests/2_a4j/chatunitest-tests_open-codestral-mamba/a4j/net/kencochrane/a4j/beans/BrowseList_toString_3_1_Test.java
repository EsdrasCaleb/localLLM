package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseList_toString_3_1_Test {

    private BrowseList browseList;

    private BrowseNode node1, node2;

    @BeforeEach
    public void setUp() {
        browseList = new BrowseList();
        node1 = Mockito.mock(BrowseNode.class);
        node2 = Mockito.mock(BrowseNode.class);
        Mockito.when(node1.getBrowseName()).thenReturn("Node1");
        Mockito.when(node1.getBrowseId()).thenReturn("1");
        Mockito.when(node2.getBrowseName()).thenReturn("Node2");
        Mockito.when(node2.getBrowseId()).thenReturn("2");
        ArrayList<BrowseNode> nodes = new ArrayList<>();
        nodes.add(node1);
        nodes.add(node2);
        browseList.setBrowseNode(nodes.toArray(new BrowseNode[0]));
    }

    @Test
    public void testToString() {
        String expected = "# of nodes = 2\n" + "Name: Node1\n" + "ID: 1\n" + "Name: Node2\n" + "ID: 2\n";
        String actual = browseList.toString();
        assertEquals(expected, actual);
    }
}
