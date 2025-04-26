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
    void testGetListId() {
        Lists lists = mock(Lists.class);
        when(lists.getListId(0)).thenReturn("test");
        String result = lists.getListId(0);
        assertEquals("test", result);
    }
}
