package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseNode_toString_13_1_Test {

    @Test
    public void testToStringMethod() throws IllegalAccessException {
        // Mock the BrowseNode object
        BrowseNode browseNode = mock(BrowseNode.class);
        // Set up the expected behavior of the getBrowseId(), getBrowseName(), and getMode() methods
        when(browseNode.getBrowseId()).thenReturn("123");
        when(browseNode.getBrowseName()).thenReturn("Example Node");
        when(browseNode.getMode()).thenReturn("Public");
        // Call the toString() method on the mock object
        String result = browseNode.toString();
        // Assert that the returned string matches the expected output
        assertEquals("123 - Example Node -- Public", result);
    }
}
