package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Authors_getAuthor_3_0_Test {

    private Authors authors;

    @BeforeEach
    public void setUp() {
        authors = new Authors();
        authors.author = Mockito.mock(ArrayList.class);
        // Set up mock expectations
        when(authors.author.size()).thenReturn(5);
        when(authors.author.get(0)).thenReturn("Author1");
        when(authors.author.get(1)).thenReturn("Author2");
        when(authors.author.get(2)).thenReturn("Author3");
        when(authors.author.get(3)).thenReturn("Author4");
        when(authors.author.get(4)).thenReturn("Author5");
    }

    @Test
    public void testGetAuthor() {
        assertEquals("Author1", authors.getAuthor(0));
        assertEquals("Author2", authors.getAuthor(1));
        assertEquals("Author3", authors.getAuthor(2));
        assertEquals("Author4", authors.getAuthor(3));
        assertEquals("Author5", authors.getAuthor(4));
    }
}
