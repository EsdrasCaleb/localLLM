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

class Authors_toString_4_0_Test {

    @InjectMocks
    private Authors authors;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWithAuthors() {
        String[] authorNames = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorNames);
        String expected = "# of Authors = 3\nAuthor - Author1\nAuthor - Author2\nAuthor - Author3\n";
        String result = authors.toString();
        assertEquals(expected, result);
    }

    @Test
    void testToStringWithNoAuthors() {
        authors.setAuthor(new String[] {});
        String expected = "Authors is null or size 0\n";
        String result = authors.toString();
        assertEquals(expected, result);
    }

    @Test
    void testToStringWithNullAuthors() {
        authors.setAuthor(null);
        String expected = "Authors is null or size 0\n";
        String result = authors.toString();
        assertEquals(expected, result);
    }
}
