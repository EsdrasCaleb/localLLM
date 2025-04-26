package net.kencochrane.a4j.DAO;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.*;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;
import java.util.ArrayList;

public class Product_getProduct_0_2_Test {

    @Test
    public void testGetProduct() {
        Product product = new Product();
        assertEquals(null, product.getProduct("asin", "offer", "page"));
    }
}
