package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseNode_toString_13_0_Test {

    private BrowseNode node;

    @BeforeEach
    void setUp() {
        node = new BrowseNode();
    }

    @Test
    void testToString_emptyValues() {
        String expected = "";
        node.setBrowseId("");
        node.setBrowseName("");
        node.setMode("");
        String actual = node.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToString_validValues() {
        node.setBrowseId("123");
        node.setBrowseName("Test Node");
        node.setMode("READ");
        String expected = "123 - Test Node -- READ";
        String actual = node.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToString_nullValues() {
        node.setBrowseId(null);
        node.setBrowseName(null);
        node.setMode(null);
        String expected = "";
        String actual = node.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToString_withSpaces() {
        node.setBrowseId("  123  ");
        node.setBrowseName("  Test Node  ");
        node.setMode("  READ  ");
        String expected = "  123  -  Test Node  --  READ  ";
        String actual = node.toString();
        assertEquals(expected, actual);
    }
}
