package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Authors_toString_4_0_Test {

    private Authors authors;

    @BeforeEach
    void setUp() {
        authors = new Authors();
    }

    @Test
    void testToString_WithAuthors() {
        String[] authorNames = { "Author One", "Author Two", "Author Three" };
        authors.setAuthor(authorNames);
        String expectedOutput = "# of Authors = 3\n" + "Author - Author One\n" + "Author - Author Two\n" + "Author - Author Three\n";
        assertEquals(expectedOutput, authors.toString());
    }

    @Test
    void testToString_EmptyAuthors() {
        authors.setAuthor(new String[0]);
        String expectedOutput = "Authors is null or size 0\n";
        assertEquals(expectedOutput, authors.toString());
    }

    @Test
    void testToString_NullAuthors() throws Exception {
        // Using reflection to set the author field to null
        java.lang.reflect.Field field = Authors.class.getDeclaredField("author");
        field.setAccessible(true);
        field.set(authors, null);
        String expectedOutput = "Authors is null or size 0\n";
        assertEquals(expectedOutput, authors.toString());
    }
}
