// Test method
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Authors_toString_4_0_Test implements Serializable {

    @Mock
    private Authors authors;

    @Test
    public void testToString() {
        // Arrange
        String[] expectedAuthors = { "Author1", "Author2", "Author3" };
        ArrayList<String> expectedAuthorsList = new ArrayList<>(Arrays.asList(expectedAuthors));
        when(authors.getAuthor()).thenReturn(expectedAuthors);
        when(authors.getAuthorsArray()).thenReturn(expectedAuthorsList);
        // Act
        String result = authors.toString();
        // Assert
        Assertions.assertEquals("# of Authors = 3\n" + "Author - Author1\n" + "Author - Author2\n" + "Author - Author3", result);
    }
}
