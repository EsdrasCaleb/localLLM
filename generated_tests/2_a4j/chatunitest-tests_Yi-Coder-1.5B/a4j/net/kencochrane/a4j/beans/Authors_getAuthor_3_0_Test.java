package net.kencochrane.a4j.beans;

import java.lang.reflect.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Authors_getAuthor_3_0_Test {

    @Test
    void getAuthorTest() {
        Authors authors = new Authors();
        String[] authorsArray = { "author1", "author2", "author3", "author4", "author5" };
        authors.setAuthor(authorsArray);
        assertEquals("author1", authors.getAuthor(0));
        assertEquals("author2", authors.getAuthor(1));
        assertEquals("author3", authors.getAuthor(2));
        assertEquals("author4", authors.getAuthor(3));
        assertEquals("author5", authors.getAuthor(4));
    }
}
