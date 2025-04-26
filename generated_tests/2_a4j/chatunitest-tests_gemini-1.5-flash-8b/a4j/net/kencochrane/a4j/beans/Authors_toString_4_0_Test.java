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

    private Authors authors;

    @BeforeEach
    void setUp() {
        authors = new Authors();
    }

    @Test
    void testToString_emptyAuthorList() {
        assertEquals("Authors is null or size 0\n", authors.toString());
    }

    @Test
    void testToString_nonEmptyAuthorList() {
        String[] authorsArray = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorsArray);
        String expectedOutput = "# of Authors = 3\n" + "Author - Author1\n" + "Author - Author2\n" + "Author - Author3\n";
        assertEquals(expectedOutput, authors.toString());
    }
}
