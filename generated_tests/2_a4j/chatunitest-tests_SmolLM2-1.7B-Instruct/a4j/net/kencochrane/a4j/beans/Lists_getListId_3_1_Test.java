// Test method
package net.kencochrane.a4j.beans;

import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;

@ExtendWith(MockitoExtension.class)
public class Lists_getListId_3_1_Test {

    @Mock
    private Lists lists;

    @Test
    void testGetListId() {
        // Arrange
        when(lists.getListId(0)).thenReturn("list1");
        // Act
        String listId = lists.getListId(0);
        // Assert
        assertEquals("list1", listId);
    }
}
