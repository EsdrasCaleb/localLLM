// Test class
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Lists_getListId_3_4_Test {

    @Mock
    private ArrayList<String> mockLists;

    @InjectMocks
    private Lists lists;

    @Test
    public void testGetListId() {
        String[] listIds = { "list1", "list2", "list3" };
        lists.setListId(listIds);
        when(mockLists.size()).thenReturn(listIds.length);
        when(mockLists.get(0)).thenReturn(listIds[0]);
        String result = lists.getListId(0);
        assertEquals(listIds[0], result);
    }
}
