package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_toString_13_0_Test {

    private BrowseNode browseNode;

    @BeforeEach
    public void setUp() {
        browseNode = new BrowseNode();
    }

    @Test
    public void testToString_withNullValues() {
        // Arrange
        browseNode.setBrowseId(null);
        browseNode.setBrowseName(null);
        browseNode.setMode(null);
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("null - null -- null", result);
    }

    @Test
    public void testToString_withEmptyValues() {
        // Arrange
        browseNode.setBrowseId("");
        browseNode.setBrowseName("");
        browseNode.setMode("");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals(" -  -- ", result);
    }

    @Test
    public void testToString_withValidValues() {
        // Arrange
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("SampleNode");
        browseNode.setMode("Active");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("123 - SampleNode -- Active", result);
    }

    @Test
    public void testToString_withWhitespaceValues() {
        // Arrange
        browseNode.setBrowseId("   ");
        browseNode.setBrowseName("   ");
        browseNode.setMode("   ");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("    -    --    ", result);
    }
}
