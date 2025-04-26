package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseList_toString_3_2_Test {

    @Test
    public void testToString() {
        // Arrange
        BrowseList browseList = mock(BrowseList.class);
        when(browseList.getBrowseNodeList()).thenReturn(new ArrayList<>());
        // Act
        String result = browseList.toString();
        // Assert
        assertEquals("No nodes\n", result);
    }
}
