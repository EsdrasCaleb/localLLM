package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseList_toString_3_2_Test {

    @Test
    public void testToString() {
        // Given
        BrowseList browseList = new BrowseList();
        BrowseNode mockNode = mock(BrowseNode.class);
        when(mockNode.getBrowseName()).thenReturn("testName");
        when(mockNode.getBrowseId()).thenReturn("testId");
        browseList.setBrowseNode(new BrowseNode[] { mockNode });
        // When
        String result = browseList.toString();
        // Then
        assertEquals("Name: testName\nID: testId\n", result);
    }

    @Test
    public void testToStringNoNodes() {
        // Given
        BrowseList browseList = new BrowseList();
        // When
        String result = browseList.toString();
        // Then
        assertEquals("No nodes\n", result);
    }
}
