package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Authors_getAuthor_3_2_Test {

    @InjectMocks
    private Authors authors;

    @Mock
    private ArrayList<String> authorList;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        authorList.add("Alice");
        authorList.add("Bob");
        authorList.add("Charlie");
    }

    @Test
    public void getAuthorTest() {
        // Arrange
        when(authors.getAuthor(1)).thenReturn("Bob");
        // Act
        String authorName = authors.getAuthor(1);
        // Assert
        assertEquals("Bob", authorName);
    }
}
