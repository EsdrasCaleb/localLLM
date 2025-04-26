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
}
