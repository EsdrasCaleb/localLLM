package net.kencochrane.a4j.DAO;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.ShoppingCart;
import net.kencochrane.a4j.beans.ShoppingCartResponse;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.file.FileUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

// Focal class
public class // Test method
Cart_clearCart_2_2_Test {

    // Focal method
    public ShoppingCart clearCart(String hmac, String cartId) {
        Query query = new Query();
        FileUtil fileUtil = new FileUtil();
        JOXBeanInputStream joxIn = null;
        ShoppingCart shoppingCart = null;
        String queryString = query.ClearCart(cartId, hmac);
        // log.debug("queryString = " + queryString);
        File file = fileUtil.downloadCart(queryString);
        if (file != null) {
            // log.debug("file not null");
            try {
                FileInputStream fin = new FileInputStream(file);
                joxIn = new JOXBeanInputStream(fin);
                ShoppingCartResponse cartBean = (ShoppingCartResponse) joxIn.readObject(ShoppingCartResponse.class);
                joxIn.close();
                fin.close();
                if (cartBean != null && cartBean.getShoppingCart() != null) {
                    shoppingCart = cartBean.getShoppingCart();
                } else {
                    System.out.println("CartBean is null !");
                }
            } catch (FileNotFoundException fnfe) {
                // error
                // log.error(fnfe.toString());
                fnfe.printStackTrace();
            } catch (IOException e) {
                // log.error(e.toString());
                e.printStackTrace();
            }
        }
        return shoppingCart;
    }
}
