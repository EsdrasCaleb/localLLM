package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Authors_getAuthor_3_0_Test {

    private Authors authors;

    @BeforeEach
    void setUp() {
        authors = new Authors();
    }

    @Test
    void testGetAuthorNegativeIndex() throws Exception {
        String[] authorArray = { "Author1", "Author2", "Author3" };
        authors.setAuthor(authorArray);
        assertNull(authors.getAuthor(-1));
    }
}
