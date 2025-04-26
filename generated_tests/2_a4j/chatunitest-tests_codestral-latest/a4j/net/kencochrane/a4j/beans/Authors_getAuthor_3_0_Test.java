package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Authors_getAuthor_3_0_Test {

    @InjectMocks
    private Authors authors;

    @Mock
    private ArrayList<String> author;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetAuthorValidIndex() {
        when(author.size()).thenReturn(3);
        when(author.get(1)).thenReturn("Author2");
        String result = authors.getAuthor(1);
        assertEquals("Author2", result);
    }

    @Test
    void testGetAuthorInvalidIndex() {
        when(author.size()).thenReturn(3);
        String result = authors.getAuthor(3);
        assertNull(result);
    }

    @Test
    void testGetAuthorNegativeIndex() {
        when(author.size()).thenReturn(3);
        String result = authors.getAuthor(-1);
        assertNull(result);
    }
}
