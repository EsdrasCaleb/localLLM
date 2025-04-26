package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
class Authors_getAuthor_3_2_Test {

    @Mock
    private Authors authors;

    @Test
    void testGetAuthor() {
        // Arrange
        String[] author = { "John Doe", "Jane Doe", "Bob Smith" };
        int index = 0;
        // Act
        String result = authors.getAuthor(index);
        // Assert
        assert result.equals("John Doe");
    }
}
