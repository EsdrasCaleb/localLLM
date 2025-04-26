package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
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
    void setUp() {
        authors = new Authors();
    }

    @Test
    void testGetAuthorWithinBounds() throws Exception {
        String[] authorArray = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorArray);
        assertEquals("Author2", authors.getAuthor(1));
    }

    @Test
    void testGetAuthorOutOfBounds() throws Exception {
        String[] authorArray = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorArray);
        assertNull(authors.getAuthor(3));
    }

    @Test
    void testGetAuthorEmptyList() throws Exception {
        assertNull(authors.getAuthor(0));
    }

    @Test
    void testGetAuthorNegativeIndex() throws Exception {
        String[] authorArray = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorArray);
        assertNull(authors.getAuthor(-1));
    }

    @Test
    void testGetAuthorListWithOneElement() throws Exception {
        String[] authorArray = { "Author1" };
        authors.setAuthor(authorArray);
        assertEquals("Author1", authors.getAuthor(0));
    }

    @Test
    void testNullAuthorList() throws Exception {
        Field authorField = Authors.class.getDeclaredField("author");
        authorField.setAccessible(true);
        authorField.set(authors, null);
        assertNull(authors.getAuthor(0));
    }
}
