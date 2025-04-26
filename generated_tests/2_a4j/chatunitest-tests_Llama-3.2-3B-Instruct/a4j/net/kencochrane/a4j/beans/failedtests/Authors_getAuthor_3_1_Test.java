package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Authors_getAuthor_3_1_Test {

    @Test
    public void testGetAuthor_InValidIndex_ReturnsNull() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>();
        String result = authors.getAuthor(-1);
        assertNull(result);
    }

    @Test
    public void testGetAuthor_EmptyList_ReturnsNull() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>();
        String result = authors.getAuthor(0);
        assertNull(result);
    }

    @Test
    public void testGetAuthor_ValidIndex_ReturnsAuthor() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>();
        authors.author.add("John");
        authors.author.add("Jane");
        authors.author.add("Bob");
        String result = authors.getAuthor(1);
        assertEquals("Jane", result);
    }

    @Test
    public void testGetAuthor_ValidIndex_ReturnsNull() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>();
        authors.author.add("John");
        authors.author.add("Jane");
        authors.author.add("Bob");
        String result = authors.getAuthor(3);
        assertNull(result);
    }
}
