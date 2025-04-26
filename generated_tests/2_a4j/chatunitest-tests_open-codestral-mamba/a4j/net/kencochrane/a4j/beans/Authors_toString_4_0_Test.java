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

public class Authors_toString_4_0_Test {

    private Authors authors;

    @BeforeEach
    public void setUp() {
        authors = Mockito.mock(Authors.class);
    }

    @Test
    public void testToString() {
        ArrayList<String> authorsList = new ArrayList<>(Arrays.asList("Author 1", "Author 2", "Author 3"));
        Mockito.when(authors.getAuthorsArray()).thenReturn(authorsList);
        String expected = "# of Authors = 3\nAuthor - Author 1\nAuthor - Author 2\nAuthor - Author 3\n";
        String actual = authors.toString();
        assertEquals(expected, actual);
    }
}
