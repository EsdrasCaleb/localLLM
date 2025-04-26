package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseNode_toString_13_0_Test {

    @Test
    void testToString() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Test");
        node.setMode("TestMode");
        String expected = "123 - Test -- TestMode";
        assertEquals(expected, node.toString());
    }
}
