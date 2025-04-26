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

public class Authors_getAuthor_3_0_Test {

    @Test
    public void testGetAuthor() {
        // Arrange
        Authors authors = new Authors();
        String[] authorNames = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorNames);
        // Act
        String result = authors.getAuthor(1);
        // Assert
        assertEquals("Author2", result);
    }
}
