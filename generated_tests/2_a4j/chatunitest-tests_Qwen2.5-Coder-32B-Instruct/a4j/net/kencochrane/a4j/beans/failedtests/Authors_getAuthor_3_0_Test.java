package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Authors_getAuthor_3_0_Test {

    @InjectMocks
    private Authors authors;

    @BeforeEach
    public void setUp() throws IllegalAccessException, NoSuchFieldException {
        MockitoAnnotations.openMocks(this);
        Field authorField = Authors.class.getDeclaredField("author");
        authorField.setAccessible(true);
        authorField.set(authors, new ArrayList<>());
    }

    @Test
    public void testGetAuthor_ValidIndex() throws Exception {
        // Arrange
        ArrayList<String> authorsList = new ArrayList<>();
        authorsList.add("Author1");
        authorsList.add("Author2");
        Field authorField = Authors.class.getDeclaredField("author");
        authorField.setAccessible(true);
        authorField.set(authors, authorsList);
        // Act
        String result = authors.getAuthor(1);
        // Assert
        assertEquals("Author2", result);
    }

    @Test
    public void testGetAuthor_IndexOutOfBounds() {
        // Arrange
        ArrayList<String> authorsList = new ArrayList<>();
        authorsList.add("Author1");
        try {
            Field authorField = Authors.class.getDeclaredField("author");
            authorField.setAccessible(true);
            authorField.set(authors, authorsList);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception should not be thrown: " + e.getMessage());
        }
        // Act & Assert
        assertThrows(IndexOutOfBoundsException.class, () -> authors.getAuthor(2));
    }

    @Test
    public void testGetAuthor_EmptyList() {
        // Arrange
        ArrayList<String> authorsList = new ArrayList<>();
        try {
            Field authorField = Authors.class.getDeclaredField("author");
            authorField.setAccessible(true);
            authorField.set(authors, authorsList);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception should not be thrown: " + e.getMessage());
        }
        // Act & Assert
        assertNull(authors.getAuthor(0));
    }
}
