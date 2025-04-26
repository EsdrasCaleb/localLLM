package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Lists_getListId_3_0_Test {

    @Test
    public void testGetListId_WithinBounds_ReturnsElement() {
        Lists lists = new Lists();
        ArrayList<String> listsMock = Mockito.mock(ArrayList.class);
        when(lists.getListsArray()).thenReturn(listsMock);
        String[] newListId = { "id1", "id2", "id3" };
        lists.setListId(newListId);
        assertEquals(newListId[0], lists.getListId(0));
    }

    @Test
    public void testGetListId_OutOfBounds_ReturnsNull() {
        Lists lists = new Lists();
        ArrayList<String> listsMock = Mockito.mock(ArrayList.class);
        when(lists.getListsArray()).thenReturn(listsMock);
        lists.setListId(new String[] { "id1", "id2" });
        assertNull(lists.getListId(3));
    }

    @Test
    public void testGetListId_EmptyList_ReturnsNull() {
        Lists lists = new Lists();
        ArrayList<String> listsMock = Mockito.mock(ArrayList.class);
        when(lists.getListsArray()).thenReturn(listsMock);
        assertNull(lists.getListId(0));
    }

    @Test
    public void testGetListId_NullListsArray_ThrowsNullPointerException() {
        Lists lists = new Lists();
        assertThrows(NullPointerException.class, () -> lists.getListId(0));
    }
}
