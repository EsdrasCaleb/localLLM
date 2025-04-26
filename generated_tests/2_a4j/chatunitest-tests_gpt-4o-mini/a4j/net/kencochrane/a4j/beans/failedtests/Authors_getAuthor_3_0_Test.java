package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.util.ArrayList;
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
    void testGetAuthor_ValidIndex() {
        // Arrange
        String[] names = { "Author1", "Author2", "Author3" };
        authors.setAuthor(names);
        // Act
        // Valid index
        String result = authors.getAuthor(1);
        // Assert
        assertEquals("Author2", result);
    }

    @Test
    void testGetAuthor_IndexOutOfBounds() {
        // Arrange
        String[] names = { "Author1", "Author2", "Author3" };
        authors.setAuthor(names);
        // Act
        // Index out of bounds
        String result = authors.getAuthor(3);
        // Assert
        assertNull(result);
    }

    @Test
    void testGetAuthor_NegativeIndex() {
        // Arrange
        String[] names = { "Author1", "Author2", "Author3" };
        authors.setAuthor(names);
        // Act
        // Negative index
        String result = authors.getAuthor(-1);
        // Assert
        assertNull(result);
    }

    @Test
    void testGetAuthor_EmptyList() {
        // Arrange
        // Empty list
        authors.setAuthor(new String[0]);
        // Act
        // Valid index but list is empty
        String result = authors.getAuthor(0);
        // Assert
        assertNull(result);
    }
}
