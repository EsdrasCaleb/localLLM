package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Authors_toString_4_0_Test {

    @Test
    public void testToStringWithEmptyList() {
        Authors authors = new Authors();
        assertEquals("# of Authors = 0\n", authors.toString());
    }

    @Test
    public void testToStringWithSingleElementList() {
        Authors authors = new Authors();
        authors.setAuthor(new String[] { "John Doe" });
        assertEquals("# of Authors = 1\nAuthor - John Doe\n", authors.toString());
    }

    @Test
    public void testToStringWithMultipleElementsList() {
        Authors authors = new Authors();
        authors.setAuthor(new String[] { "John Doe", "Jane Smith", "Alice Johnson" });
        assertEquals("# of Authors = 3\nAuthor - John Doe\nAuthor - Jane Smith\nAuthor - Alice Johnson\n", authors.toString());
    }
}
