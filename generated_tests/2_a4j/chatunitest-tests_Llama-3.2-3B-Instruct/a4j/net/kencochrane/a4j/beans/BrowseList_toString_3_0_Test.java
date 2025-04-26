package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseList_toString_3_0_Test {

    @Test
    public void testToString_EmptyList() {
        BrowseList list = new BrowseList();
        String result = list.toString();
        assertNotNull(result);
        assertEquals("No nodes", result);
    }

    @Test
    public void testToString_SingleNode() {
        BrowseList list = new BrowseList();
        BrowseNode node = new BrowseNode();
        node.setBrowseName("Test");
        node.setBrowseId("123");
        list.setBrowseNode(new BrowseNode[] { node });
        String result = list.toString();
        assertNotNull(result);
        assertEquals("Name: Test\nID: 123\n", result);
    }

    @Test
    public void testToString_MultipleNodes() {
        BrowseList list = new BrowseList();
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseName("Test1");
        node1.setBrowseId("123");
        BrowseNode node2 = new BrowseNode();
        node2.setBrowseName("Test2");
        node2.setBrowseId("456");
        BrowseNode[] nodes = new BrowseNode[] { node1, node2 };
        list.setBrowseNode(nodes);
        String result = list.toString();
        assertNotNull(result);
        assertEquals("Name: Test1\nID: 123\nName: Test2\nID: 456\n", result);
    }
}
