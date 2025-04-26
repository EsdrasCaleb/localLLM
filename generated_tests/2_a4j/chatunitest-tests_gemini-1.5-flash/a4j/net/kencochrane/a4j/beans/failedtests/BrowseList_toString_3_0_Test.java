package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseList_toString_3_0_Test {

    @Test
    void testToString_emptyList() {
        BrowseList browseList = new BrowseList();
        assertEquals("No nodes\n", browseList.toString());
    }

    @Test
    void testToString_nonEmptyList() {
        BrowseNode node1 = Mockito.mock(BrowseNode.class);
        Mockito.when(node1.getBrowseName()).thenReturn("Node 1");
        Mockito.when(node1.getBrowseId()).thenReturn("ID1");
        BrowseNode node2 = Mockito.mock(BrowseNode.class);
        Mockito.when(node2.getBrowseName()).thenReturn("Node 2");
        Mockito.when(node2.getBrowseId()).thenReturn("ID2");
        BrowseList browseList = new BrowseList();
        try {
            Field nodesField = BrowseList.class.getDeclaredField("nodes");
            nodesField.setAccessible(true);
            ArrayList<BrowseNode> nodes = new ArrayList<>(Arrays.asList(node1, node2));
            nodesField.set(browseList, nodes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set nodes field: " + e.getMessage());
        }
        String expected = "# of nodes = 2\n" + "Name: Node 1\n" + "ID: ID1\n" + "Name: Node 2\n" + "ID: ID2\n";
        assertEquals(expected, browseList.toString());
    }

    @Test
    void testToString_nullList() {
        BrowseList browseList = new BrowseList();
        try {
            Field nodesField = BrowseList.class.getDeclaredField("nodes");
            nodesField.setAccessible(true);
            nodesField.set(browseList, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set nodes field: " + e.getMessage());
        }
        assertEquals("No nodes\n", browseList.toString());
    }

    @Test
    void testToString_listWithNullNode() {
        BrowseNode node1 = Mockito.mock(BrowseNode.class);
        Mockito.when(node1.getBrowseName()).thenReturn("Node 1");
        Mockito.when(node1.getBrowseId()).thenReturn("ID1");
        BrowseList browseList = new BrowseList();
        try {
            Field nodesField = BrowseList.class.getDeclaredField("nodes");
            nodesField.setAccessible(true);
            ArrayList<BrowseNode> nodes = new ArrayList<>(Arrays.asList(node1, null));
            nodesField.set(browseList, nodes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set nodes field: " + e.getMessage());
        }
        String expected = "# of nodes = 2\n" + "Name: Node 1\n" + "ID: ID1\n";
        assertEquals(expected, browseList.toString());
    }

    // Dummy BrowseNode class for compilation
    static class BrowseNode {

        public String getBrowseName() {
            return "";
        }

        public String getBrowseId() {
            return "";
        }
    }
}
