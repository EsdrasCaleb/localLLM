package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Authors_toString_4_0_Test {

    @Test
    public void testToString_EmptyList_ReturnsCorrectMessage() {
        Authors authors = new Authors();
        String result = authors.toString();
        assertEquals("Authors is null or size 0", result);
    }

    @Test
    public void testToString_NullList_ReturnsCorrectMessage() {
        Authors authors = new Authors();
        authors.author = null;
        String result = authors.toString();
        assertEquals("Authors is null or size 0", result);
    }

    @Test
    public void testToString_SingleAuthor_ReturnsCorrectMessage() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>(Arrays.asList("John Doe"));
        String result = authors.toString();
        assertEquals("Author - John Doe", result);
    }

    @Test
    public void testToString_MultipleAuthors_ReturnsCorrectMessage() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>(Arrays.asList("John Doe", "Jane Doe"));
        String result = authors.toString();
        assertEquals("Author - John Doe\nAuthor - Jane Doe", result);
    }

    @Test
    public void testToString_MultipleAuthorsWithSpaces_ReturnsCorrectMessage() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>(Arrays.asList("John Doe", "Jane Doe Smith"));
        String result = authors.toString();
        assertEquals("Author - John Doe\nAuthor - Jane Doe Smith", result);
    }
}
