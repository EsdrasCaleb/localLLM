package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Lists_getListId_3_3_Test {

    @Mock
    private Lists lists;

    @BeforeEach
    public void setUp() {
        lists = mock(Lists.class);
    }

    @Test
    public void testGetListIdWithValidIndex() throws Exception {
        // Arrange
        lists.lists = new ArrayList<>();
        lists.lists.add("Element1");
        lists.lists.add("Element2");
        int index = 1;
        when(lists.getListId(index)).thenReturn("Element2");
        // Act
        String result = lists.getListId(index);
        // Assert
        assertEquals("Element2", result);
    }

    @Test
    public void testGetListIdWithInvalidIndex() throws Exception {
        // Arrange
        lists.lists = new ArrayList<>();
        lists.lists.add("Element1");
        lists.lists.add("Element2");
        int index = 3;
        when(lists.getListId(index)).thenReturn(null);
        // Act
        String result = lists.getListId(index);
        // Assert
        assertNull(result);
    }
}
