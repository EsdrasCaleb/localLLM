package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.io.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BrowseNode_printNode_7_0_Test {

    @Test
    void testPrintNode_noSubNodes() throws NoSuchFieldException, IllegalAccessException {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        // Redirect System.out to capture output
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        node.printNode();
        System.out.flush();
        System.setOut(old);
        assertEquals("123 - Root\r\n", baos.toString());
    }

    @Test
    void testPrintNode_withSubNodes() throws NoSuchFieldException, IllegalAccessException {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        BrowseNode subNode1 = new BrowseNode();
        subNode1.setBrowseId("456");
        subNode1.setBrowseName("Child 1");
        BrowseNode subNode2 = new BrowseNode();
        subNode2.setBrowseId("789");
        subNode2.setBrowseName("Child 2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(subNode1);
        subNodes.add(subNode2);
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(node, subNodes);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        node.printNode();
        System.out.flush();
        System.setOut(old);
        assertEquals("123 - Root\r\n  -- # of subNodes 2 -- \r\n    456 - Child 1\r\n    789 - Child 2\r\n", baos.toString());
    }

    @Test
    void testPrintNode_emptySubNodes() throws NoSuchFieldException, IllegalAccessException {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(node, new ArrayList<>());
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        node.printNode();
        System.out.flush();
        System.setOut(old);
        assertEquals("123 - Root\r\n", baos.toString());
    }

    @Test
    void testPrintNode_nullSubNodes() throws NoSuchFieldException, IllegalAccessException {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(node, null);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        node.printNode();
        System.out.flush();
        System.setOut(old);
        assertEquals("123 - Root\r\n", baos.toString());
    }
}
