package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Authors_getAuthor_3_2_Test {

    @Test
    public void testGetAuthor() {
        Authors authors = new Authors();
        String[] authorNames = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorNames);
        Assertions.assertEquals("Author1", authors.getAuthor(0));
        Assertions.assertEquals("Author2", authors.getAuthor(1));
        Assertions.assertEquals("Author3", authors.getAuthor(2));
        Assertions.assertNull(authors.getAuthor(3));
    }
}
