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
    public void testAddSubNode_EmptyList() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode = new BrowseNode();
        node.addSubNode(subNode);
        assertEquals(1, node.getSubNodes().size());
    }

    @Test
    public void testAddSubNode_NonEmptyList() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode1 = new BrowseNode();
        BrowseNode subNode2 = new BrowseNode();
        node.addSubNode(subNode1);
        node.addSubNode(subNode2);
        assertEquals(2, node.getSubNodes().size());
    }

    @Test
    public void testAddSubNode_NullNode() {
        BrowseNode node = new BrowseNode();
        assertThrows(NullPointerException.class, () -> node.addSubNode(null));
    }

    @Test
    public void testAddSubNode_DuplicateNode() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode = new BrowseNode();
        node.addSubNode(subNode);
        node.addSubNode(subNode);
        assertEquals(1, node.getSubNodes().size());
    }

    @Test
    public void testAddSubNode_MultipleNodes() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode1 = new BrowseNode();
        BrowseNode subNode2 = new BrowseNode();
        BrowseNode subNode3 = new BrowseNode();
        node.addSubNode(subNode1);
        node.addSubNode(subNode2);
        node.addSubNode(subNode3);
        assertEquals(3, node.getSubNodes().size());
    }

    @Test
    public void testAddSubNode_MultipleNodesWithSameNode() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode = new BrowseNode();
        node.addSubNode(subNode);
        node.addSubNode(subNode);
        node.addSubNode(subNode);
        assertEquals(1, node.getSubNodes().size());
    }
}
