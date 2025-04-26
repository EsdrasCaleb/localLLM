package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;

public class Items_toString_3_0_Test {

    @InjectMocks
    private Items items;

    @Mock
    private ArrayList<Item> mockItems;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToStringWithItems() {
        Item item1 = new Item();
        Item item2 = new Item();
        when(mockItems.size()).thenReturn(2);
        when(mockItems.get(0)).thenReturn(item1);
        when(mockItems.get(1)).thenReturn(item2);
        String expected = item1.toString() + item2.toString();
        String result = items.toString();
        assertEquals(expected, result);
    }

    @Test
    public void testToStringWithNullItems() {
        when(mockItems.size()).thenReturn(0);
        String result = items.toString();
        assertEquals("", result);
    }

    @Test
    public void testToStringWithNullItemInList() {
        Item item1 = new Item();
        when(mockItems.size()).thenReturn(2);
        when(mockItems.get(0)).thenReturn(item1);
        when(mockItems.get(1)).thenReturn(null);
        String expected = item1.toString();
        String result = items.toString();
        assertEquals(expected, result);
    }
}
