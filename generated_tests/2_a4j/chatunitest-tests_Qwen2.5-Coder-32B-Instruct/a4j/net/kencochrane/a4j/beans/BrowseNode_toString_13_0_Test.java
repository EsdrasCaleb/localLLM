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
    public void testToString_AllFieldsSet() {
        // Arrange
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("Test Node");
        browseNode.setMode("ACTIVE");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("123 - Test Node -- ACTIVE", result);
    }

    @Test
    public void testToString_BrowseIdNull() {
        // Arrange
        browseNode.setBrowseName("Test Node");
        browseNode.setMode("ACTIVE");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("null - Test Node -- ACTIVE", result);
    }

    @Test
    public void testToString_BrowseNameNull() {
        // Arrange
        browseNode.setBrowseId("123");
        browseNode.setMode("ACTIVE");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("123 - null -- ACTIVE", result);
    }

    @Test
    public void testToString_ModeNull() {
        // Arrange
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("Test Node");
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("123 - Test Node -- null", result);
    }

    @Test
    public void testToString_AllFieldsNull() {
        // Act
        String result = browseNode.toString();
        // Assert
        assertEquals("null - null -- null", result);
    }
}
