package net.kencochrane.a4j.beans;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseNode_printNode_7_0_Test {

    @Test
    void printNode_noSubNodes() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        String expectedOutput = "123 - Root";
        String actualOutput = captureOutput(() -> node.printNode());
        assertEquals(expectedOutput, actualOutput.trim());
    }

    @Test
    void printNode_withSubNodes() {
        BrowseNode parent = new BrowseNode();
        parent.setBrowseId("1");
        parent.setBrowseName("Parent");
        BrowseNode child1 = new BrowseNode();
        child1.setBrowseId("101");
        child1.setBrowseName("Child 1");
        BrowseNode child2 = new BrowseNode();
        child2.setBrowseId("102");
        child2.setBrowseName("Child 2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>(Arrays.asList(child1, child2));
        parent.setSubNodes(subNodes);
        String expectedOutput = "1 - Parent  -- # of subNodes 2 -- \n    101 - Child 1    102 - Child 2";
        String actualOutput = captureOutput(() -> parent.printNode());
        assertEquals(expectedOutput.replace("\n", System.lineSeparator()), actualOutput.trim());
    }

    @Test
    void printNode_withNullSubNodes() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        node.setSubNodes(null);
        String expectedOutput = "123 - Root";
        String actualOutput = captureOutput(() -> node.printNode());
        assertEquals(expectedOutput, actualOutput.trim());
    }

    @Test
    void printNode_withEmptySubNodes() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Root");
        node.setSubNodes(new ArrayList<>());
        String expectedOutput = "123 - Root";
        String actualOutput = captureOutput(() -> node.printNode());
        assertEquals(expectedOutput, actualOutput.trim());
    }

    private String captureOutput(Runnable runnable) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(baos));
        runnable.run();
        System.setOut(originalOut);
        return baos.toString().trim();
    }
}
