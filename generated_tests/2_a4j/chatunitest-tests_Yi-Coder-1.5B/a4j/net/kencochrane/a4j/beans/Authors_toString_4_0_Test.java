package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Authors_toString_4_0_Test {

    Authors authors = new Authors();

    @Test
    void testToString() {
        assertEquals("Authors is null or size 0", authors.toString());
    }

    @Test
    void testToStringWithAuthors() {
        authors.setAuthor(new String[] { "Jane Doe", "John Doe" });
        assertEquals("# of Authors = 2\nAuthor - Jane Doe\nAuthor - John Doe", authors.toString());
    }
}
