package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Authors_getAuthor_3_0_Test {

    @Mock
    private ArrayList<String> author;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetAuthor() {
        when(author.get(0)).thenReturn("John Doe");
        when(author.get(1)).thenReturn("Jane Smith");
        when(author.get(2)).thenReturn("Alice Johnson");
        Authors authors = new Authors();
        authors.setAuthor(new String[] { "John Doe", "Jane Smith", "Alice Johnson" });
        assertEquals("John Doe", authors.getAuthor(0));
        assertEquals("Jane Smith", authors.getAuthor(1));
        assertEquals("Alice Johnson", authors.getAuthor(2));
    }
}
