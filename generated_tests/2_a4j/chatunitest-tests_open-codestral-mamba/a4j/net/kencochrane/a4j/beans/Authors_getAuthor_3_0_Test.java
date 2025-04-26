package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class Authors_getAuthor_3_0_Test {

    @Mock
    private Authors authors;

    @Captor
    private ArgumentCaptor<String[]> captor;

    @Test
    void testGetAuthorWithinBounds() {
        ArrayList<String> authorList = new ArrayList<>(Arrays.asList("Author1", "Author2", "Author3"));
        when(authors.getAuthor(1)).thenReturn("Author2");
        assertEquals("Author2", authors.getAuthor(1));
    }

    @Test
    void testGetAuthorOutOfBounds() {
        ArrayList<String> authorList = new ArrayList<>(Arrays.asList("Author1", "Author2", "Author3"));
        when(authors.getAuthor(5)).thenReturn(null);
        assertNull(authors.getAuthor(5));
    }
}
