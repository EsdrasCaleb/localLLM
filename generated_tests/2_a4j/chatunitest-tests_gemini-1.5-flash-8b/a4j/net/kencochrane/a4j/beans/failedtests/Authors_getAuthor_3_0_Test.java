package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Authors_getAuthor_3_0_Test {

    private Authors authors;

    @BeforeEach
    void setUp() {
        authors = new Authors();
    }

    @Test
    void getAuthor_validIndex_returnsAuthor() {
        String[] authorsArray = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorsArray);
        String author = authors.getAuthor(1);
        assertEquals("Author2", author);
    }

    @Test
    void getAuthor_invalidIndex_returnsNull() {
        String[] authorsArray = { "Author1", "Author2" };
        authors.setAuthor(authorsArray);
        String author = authors.getAuthor(2);
        assertNull(author);
    }

    @Test
    void getAuthor_emptyArray_returnsNull() {
        String[] authorsArray = {};
        authors.setAuthor(authorsArray);
        String author = authors.getAuthor(0);
        assertNull(author);
    }

    @Test
    void getAuthor_indexIsMinusOne_returnsNull() {
        String[] authorsArray = { "Author1", "Author2" };
        authors.setAuthor(authorsArray);
        String author = authors.getAuthor(-1);
        assertNull(author);
    }

    @Test
    void getAuthor_indexIsGreaterThanSize_returnsNull() {
        String[] authorsArray = { "Author1", "Author2" };
        authors.setAuthor(authorsArray);
        String author = authors.getAuthor(2);
        assertNull(author);
    }
}
