package net.kencochrane.a4j.beans;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

@RunWith(MockitoJUnitRunner.class)
public class Authors_toString_4_0_Test {

    @InjectMocks
    private Authors authors;

    @Mock
    private Authors authorsMock;

    @Test
    public void testToString_ValidAuthors() {
        List<String> authorsList = Arrays.asList("John Doe", "Jane Doe");
        authors.setAuthor(authorsList.toArray(new String[0]));
        String expected = "# of Authors = 2\nAuthor - John Doe\nAuthor - Jane Doe";
        assertEquals(expected, authors.toString());
    }

    @Test
    public void testToString_EmptyAuthors() {
        List<String> authorsList = new ArrayList<>();
        authors.setAuthor(authorsList.toArray(new String[0]));
        String expected = "Authors is null or size 0";
        assertTrue(authors.toString().contains(expected));
    }

    @Test
    public void testToString_NullAuthors() {
        authors.setAuthor(null);
        String expected = "Authors is null or size 0";
        assertTrue(authors.toString().contains(expected));
    }
}
