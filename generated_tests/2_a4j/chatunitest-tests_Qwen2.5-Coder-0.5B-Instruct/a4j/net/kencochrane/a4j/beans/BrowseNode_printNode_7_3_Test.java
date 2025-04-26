package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_printNode_7_3_Test {

    private BrowseNode browseNode;

    @BeforeEach
    public void setUp() {
        browseNode = new BrowseNode();
    }

    @Test
    public void testPrintNode() {
        browseNode.setBrowseId("12345");
        browseNode.setBrowseName("Sample Node");
        browseNode.setMode("Read");
        browseNode.printNode();
        assertEquals("12345 - Sample Node", browseNode.getBrowseId());
        assertEquals("Sample Node", browseNode.getBrowseName());
        assertEquals(0, browseNode.getSubNodes().size());
        assertEquals("Read", browseNode.getMode());
    }
}
