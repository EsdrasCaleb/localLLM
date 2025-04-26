package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class Items_toString_3_0_Test {

    @InjectMocks
    private Items items;

    @Mock
    private ArrayList<Item> mockItems;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Injecting mockItems into the items object using reflection
        Field itemsField = Items.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(items, mockItems);
    }

    @Test
    void testToString_ItemsListIsNull() throws NoSuchFieldException, IllegalAccessException {
        Field itemsField = Items.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(items, null);
        String result = items.toString();
        assertEquals("", result);
    }

    @Test
    void testToString_ItemsListIsEmpty() {
        when(mockItems.size()).thenReturn(0);
        String result = items.toString();
        assertEquals("", result);
    }

    @Test
    void testToString_ItemsListContainsOneItem() {
        Item mockItem = mock(Item.class);
        when(mockItems.size()).thenReturn(1);
        when(mockItems.get(0)).thenReturn(mockItem);
        when(mockItem.toString()).thenReturn("Item1");
        String result = items.toString();
        assertEquals("Item1", result);
    }

    @Test
    void testToString_ItemsListContainsMultipleItems() {
        Item mockItem1 = mock(Item.class);
        Item mockItem2 = mock(Item.class);
        when(mockItems.size()).thenReturn(2);
        when(mockItems.get(0)).thenReturn(mockItem1);
        when(mockItems.get(1)).thenReturn(mockItem2);
        when(mockItem1.toString()).thenReturn("Item1");
        when(mockItem2.toString()).thenReturn("Item2");
        String result = items.toString();
        assertEquals("Item1Item2", result);
    }

    @Test
    void testToString_ItemsListContainsNullItem() {
        Item mockItem = mock(Item.class);
        when(mockItems.size()).thenReturn(2);
        when(mockItems.get(0)).thenReturn(null);
        when(mockItems.get(1)).thenReturn(mockItem);
        when(mockItem.toString()).thenReturn("Item1");
        String result = items.toString();
        assertEquals("Item1", result);
    }
}
