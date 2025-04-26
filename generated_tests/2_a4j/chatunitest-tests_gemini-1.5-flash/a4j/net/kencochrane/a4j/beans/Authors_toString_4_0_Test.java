package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Authors_toString_4_0_Test {

    private Authors authors;

    @BeforeEach
    void setUp() {
        authors = new Authors();
    }

    @Test
    void testToString_NullAuthorList() {
        // Set author to null using reflection to test the else branch.
        try {
            Field authorField = Authors.class.getDeclaredField("author");
            authorField.setAccessible(true);
            authorField.set(authors, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access author field: " + e.getMessage());
        }
        assertEquals("Authors is null or size 0\n", authors.toString());
    }

    @Test
    void testToString_EmptyAuthorList() {
        assertEquals("Authors is null or size 0\n", authors.toString());
    }

    @Test
    void testToString_SingleAuthor() {
        authors.setAuthor(new String[] { "Jane Doe" });
        assertEquals("# of Authors = 1\nAuthor - Jane Doe\n", authors.toString());
    }

    @Test
    void testToString_MultipleAuthors() {
        authors.setAuthor(new String[] { "Jane Doe", "John Smith", "Peter Jones" });
        assertEquals("# of Authors = 3\nAuthor - Jane Doe\nAuthor - John Smith\nAuthor - Peter Jones\n", authors.toString());
    }
}
