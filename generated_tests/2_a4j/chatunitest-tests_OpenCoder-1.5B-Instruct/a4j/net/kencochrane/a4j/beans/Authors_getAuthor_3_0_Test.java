package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Authors_getAuthor_3_0_Test {

    @Test
    void testGetAuthor() {
        Authors authors = new Authors();
        authors.author = new ArrayList<String>();
        authors.author.add("John Smith");
        authors.author.add("Jane Doe");
        authors.author.add("Jim Beam");
        assertEquals("Jane Doe", authors.getAuthor(1));
        assertEquals("Jim Beam", authors.getAuthor(2));
        assertNull(authors.getAuthor(3));
    }
}
